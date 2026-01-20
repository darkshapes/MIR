from diffusers import CosmosTransformer3DModel


model = CosmosTransformer3DModel()
print(type(model.transformer_blocks[0]))
for i in model.transformer_blocks[0]:
    print(type(i))
