class MbFactory:
    """Factory class to create membrane generator instances based on type.
    """

    """Registry of membrane generator classes.
    """
    __registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a membrane generator class with a given name.
        
        Args:
            name (str): The name to register the membrane generator class under.
        
        Returns:
            function: The decorator function.
        """

        def inner(subclass):
            cls.__registry[name] = subclass
            return subclass
        return inner

    @classmethod
    def create(cls, mb_type, params):
        """Create an instance of a membrane generator based on the type and parameters.
        
        Args:
            mb_type (str): The type of membrane generator to create.
            params (dict): The parameters to initialize the membrane generator.
        
        Returns:
            MbGen: An instance of the requested membrane generator.
        """
        if mb_type not in cls.__registry:
            raise ValueError(f"Membrane type '{mb_type}' is not registered. Available types: {list(cls.__registry.keys())}")
        return cls.__registry[mb_type].from_params(params)
