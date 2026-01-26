Apply
import inspect
import torch.nn as nn

def find_classes_with_module_list(module_or_class):
    """
    Finds classes in a module or a script that use nn.ModuleList.

    Args:
        module_or_class (module or class): The module or class to inspect.

    Returns:
        List of class names that use nn.ModuleList.
    """
    classes_with_module_list = []
    if isinstance(module_or_class, type):
        # If it's a class, just inspect that class
        for name, obj in inspect.getmembers(module_or_class):
            if inspect.isclass(obj) and 'nn.ModuleList' in inspect.getsource(obj):
                classes_with_module_list.append(name)
    else:
        # If it's a module, inspect all classes within it
        for name, obj in inspect.getmembers(module_or_class):
            if inspect.isclass(obj):
                if 'nn.ModuleList' in inspect.getsource(obj):
                    classes_with_module_list.append(name)

    return classes_with_module_list

# Example usage:
from torch.nn import ModuleList

# Assuming you have a module named 'y_module'
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.modules = ModuleList([nn.Linear(10, 20)])

# Find classes using nn.ModuleList
classes_with_module_list = find_classes_with_module_list(nn)
print("Classes using nn.ModuleList:", classes_with_module_list)

Classes using nn.ModuleList: ['ModuleList']
