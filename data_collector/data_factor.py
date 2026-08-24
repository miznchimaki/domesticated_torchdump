from .data_processor import (
    DefaultDataProcessor,
    OverflowCheckDataProcessor,
    FreeBenchmarkDataProcessor,
)


class DataProcessorFactory:
    _data_processor = {}

    @classmethod
    def register_processor(cls, task, processor_class):
        cls._data_processor[task] = processor_class

    @classmethod
    def create_processor(cls, config):
        cls.register_processors()
        processor_class = cls._data_processor.get(config.task)
        if not processor_class:
            raise ValueError(f"Processor not found for task: {config.task}")
        return processor_class(config)
    
    @classmethod
    def register_processors(cls):
        cls.register_processor("default", DefaultDataProcessor)
        cls.register_processor("overflow_check", OverflowCheckDataProcessor)
        cls.register_processor("free_benchmark", FreeBenchmarkDataProcessor)
