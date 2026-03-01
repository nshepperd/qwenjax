from collections import OrderedDict
from functools import partial, wraps
from typing import Callable, ParamSpec, TypeVar

import jax
import jax.extend.core
from jax._src.named_sharding import UNSPECIFIED
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, Var
from jax.interpreters import ad, batching, mlir

_label_p = jax.extend.core.Primitive('label')
def label(x: jax.Array, name: str) -> jax.Array:
    return _label_p.bind(x, name=name)

@_label_p.def_impl
def _label_impl(x: jax.Array, *, name: str):
    return x

@_label_p.def_abstract_eval
def _label_abstract_eval(x: jax.Array, *, name: str):
    return x

@partial(batching.primitive_batchers.__setitem__, _label_p)
def _label_batch(vector_arg_values, batch_axes, *, name: str):
    # Vmapping a label returns a batched output.
    return label(vector_arg_values[0], name=name), batch_axes[0]

@partial(mlir.register_lowering, _label_p) # type: ignore
def _label_lowering(ctx: mlir.LoweringRuleContext, xc, *, name: str):
    # Erase label on jit.
    return mlir.lower_fun(lambda x: x, multiple_results=False)(ctx, xc)

@partial(ad.primitive_jvps.__setitem__, _label_p)
def _label_jvp(primals, tangents, *, name: str):
    x, = primals
    t, = tangents
    y = label(x, name=name)
    return y, t

P = ParamSpec("P")
R = TypeVar("R")
def collect(fn: Callable[P, R], static_argnames=()) -> Callable[P, tuple[R, OrderedDict[str, jax.Array]]]:
    """label/collect: return extra outputs from inside a jax computation.
    
    Useful for unit testing models under jax transforms where we can't just 
    smuggle internal values out with mutability. Currently works under vmap, jit and grad.
    Maybe I'll add an option to collect the gradients too later.
    Usage example: 
        
        collect(lambda x: label(x, name='x')+1)(3) == (4, [('x', 3)])
    """
    if not jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(fn)):
        def eta(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
            return fn(*args, **kwargs)
        return partial(collect(eta, static_argnames=static_argnames), fn) # type: ignore

    def collect_jaxpr(closed: ClosedJaxpr) -> tuple[ClosedJaxpr, list[tuple[str, Var]]]:
        labels = []
        new_eqns = []
        jaxpr: Jaxpr = closed.jaxpr
        for i, eqn in enumerate(jaxpr.eqns):
            if eqn.primitive == _label_p:
                labels.append((eqn.params['name'], eqn.invars[0]))
                new_eqns.append(eqn)
            elif eqn.primitive == jax.extend.core.primitives.jit_p:
                # Recursively collect from jitted functions
                sub_closed = eqn.params['jaxpr']
                new_sub_closed, sub_labels = collect_jaxpr(sub_closed)
                sub_vars = [Var(v.aval) for (name, v) in sub_labels]
                new_params = eqn.params | {'jaxpr': new_sub_closed}
                # Let jit infer output sharding and layouts for new outputs.
                if 'out_shardings' in new_params:
                    new_params['out_shardings'] += (UNSPECIFIED,) * len(sub_vars)
                if 'out_layouts' in new_params:
                    new_params['out_layouts'] += (None,) * len(sub_vars)
                new_eqn = eqn.replace(params=new_params, outvars=eqn.outvars + sub_vars)
                labels.extend([
                    (name, v) for ((name, _), v) in zip(sub_labels, sub_vars)
                ])
                new_eqns.append(new_eqn)
            elif eqn.primitive == jax.extend.core.primitives.scan_p:
                # example:
                # { lambda ; a:i32[3]. let
                #     _:i32[] b:i32[3] = scan[
                #       _split_transpose=False
                #       jaxpr={ lambda ; c:i32[] d:i32[]. let
                #           e:i32[] = label[name=carry] c
                #           f:i32[] = convert_element_type[new_dtype=int32 weak_type=False] c
                #           g:i32[] = add f d
                #         in (g, e) }
                #       length=3
                #       linear=(False, False)
                #       num_carry=1
                #       num_consts=0
                #       reverse=False
                #       unroll=1
                #     ] 0:i32[] a
                #   in (b,) }
                sub_closed = eqn.params['jaxpr']
                new_sub_closed, sub_labels = collect_jaxpr(sub_closed)
                sub_vars = []
                for (name, v) in sub_labels:
                    aval = v.aval
                    if isinstance(aval, ShapedArray):
                        aval = aval.update(shape=(eqn.params['length'],) + aval.shape)
                        sub_vars.append(Var(aval))
                    else:
                        raise NotImplementedError(f"Cannot handle scan collected var with aval {aval}")
                new_params = eqn.params | {'jaxpr': new_sub_closed}
                new_eqn = eqn.replace(params=new_params, outvars=eqn.outvars + sub_vars)
                labels.extend([
                    (name, v) for ((name, _), v) in zip(sub_labels, sub_vars)
                ])
                new_eqns.append(new_eqn)
            else:
                new_eqns.append(eqn)
                
        new_jaxpr = jaxpr.replace(
            outvars=jaxpr.outvars + [v for (name,v) in labels],
            eqns=new_eqns,
        )
        return closed.replace(jaxpr=new_jaxpr), labels

    @wraps(fn)
    def deco(*args: P.args, **kwargs: P.kwargs) -> tuple[R, OrderedDict[str, jax.Array]]:
        static_kwargs, kwargs = ( # type: ignore
            {k: v for k, v in kwargs.items() if k in static_argnames},
            {k: v for k, v in kwargs.items() if k not in static_argnames},
        )
        leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        def flat_fn(*leaves):
            args, kwargs = jax.tree_util.tree_unflatten(treedef, leaves)
            return fn(*args, **kwargs, **static_kwargs) # type: ignore
        closed, out_shape = jax.make_jaxpr(flat_fn, return_shape=True)(*leaves)

        new_closed, labels = collect_jaxpr(closed)

        out_shapes, out_treedef = jax.tree_util.tree_flatten(out_shape)
        num_out = len(out_shapes)
        out_leaves = jax.extend.core.jaxpr_as_fun(new_closed)(*leaves)
        out_leaves, extra_leaves = out_leaves[:num_out], out_leaves[num_out:]
        collected = OrderedDict([(name, leaf) for ((name, v), leaf) in zip(labels, extra_leaves)])
        return jax.tree_util.tree_unflatten(out_treedef, out_leaves), collected
    return deco