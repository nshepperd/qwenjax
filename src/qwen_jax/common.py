from typing import Sequence

class ModulePath:
    components: tuple[str, ...]

    def __init__(self, components: Sequence[str] | str | 'ModulePath'):
        if isinstance(components, str):
            components = components.strip('.').split('.')
        elif isinstance(components, ModulePath):
            components = components.components
        self.components = tuple(components)

    def len(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, index: int) -> str:
        return self.components[index]

    def __str__(self) -> str:
        return '.'.join(self.components)

    def __truediv__(self, other: 'str | ModulePath') -> 'ModulePath':
        if isinstance(other, ModulePath):
            return ModulePath(self.components + other.components)
        elif isinstance(other, str):
            other = other.strip('.')
            return ModulePath(self.components + tuple(other.split('.')))
        else:
            raise TypeError(f"Unsupported type for path component: {type(other)}")
