from training.dataset import create_dataset

encoded = [5,2,8,1,9,4]

inputs, targets = create_dataset(encoded, block_size=4)

print("Inputs:")
for x in inputs:
    print(x)

print("\nTargets:")
for y in targets:
    print(y)