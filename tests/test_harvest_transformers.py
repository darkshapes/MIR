from mir.generate.transformers.harvest import HarvestLoop


def test_harvest():
    harvest_classes = HarvestLoop()
    harvest_classes()
