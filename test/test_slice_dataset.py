from datasets import load_dataset
from PIL import Image


def test_load_cellmap_2d():

    repo_id = "eminorhan/cellmap-2d"
    
    print(f"Attempting to load '{repo_id}' in streaming mode...")
    
    # Use streaming=True to avoid downloading the entire dataset just for a test
    ds = load_dataset(repo_id, split="train", num_proc=64)

    for i in range(len(ds)):
        print(f"Row {i}: crop name {ds[i]['crop_name']}, image shape {ds[i]['image'].size}, axis {ds[i]['axis']}, slice {ds[i]['slice']}")

    # # Fetch the very first sample from the generator
    # first_item = next(iter(dataset))
    
    # # 1. Verify all expected columns are present
    # expected_keys = {"image", "crop_name", "axis", "slice"}
    # assert expected_keys.issubset(first_item.keys()), f"Missing columns! Found: {first_item.keys()}"
    
    # # 2. Verify data types
    # assert isinstance(first_item["image"], Image.Image), f"Expected PIL Image, got {type(first_item['image'])}"
    # assert isinstance(first_item["crop_name"], str), "crop_name should be a string"
    # assert isinstance(first_item["axis"], str), "axis should be a string"
    # assert isinstance(first_item["slice"], int), "slice should be an integer"
    
    # # 3. Verify value constraints
    # assert first_item["axis"] in ["x", "y", "z"], f"Invalid axis value: {first_item['axis']}"
    # assert first_item["slice"] >= 0, f"Slice index must be >= 0, got {first_item['slice']}"
    
    # # Optional: Verify the image has actual data (width and height > 0)
    # width, height = first_item["image"].size
    # assert width > 0 and height > 0, f"Image dimensions are invalid: {width}x{height}"

    # print("\n✅ Test passed successfully!")
    # print(f"Sample Crop Name: {first_item['crop_name']}")
    # print(f"Sample Axis: {first_item['axis']}")
    # print(f"Sample Slice Index: {first_item['slice']}")
    # print(f"Sample Image Size: {width}x{height}")

    # # Iterate through the dataset and break after k rows
    # for i, row in enumerate(dataset):
    #     if i >= 100000:
    #         break
            
    #     crop_name = row["crop_name"]
    #     axis = row["axis"]
    #     slice_idx = row["slice"]
        
    #     # The image is a PIL object, so .size gives (Width, Height).
    #     # To see the standard (Height, Width) shape, we can cast it to a numpy array.
    #     pil_size = row["image"].size
        
    #     print(f"--- Row {i + 1} ---")
    #     print(f"Crop Name : {crop_name}")
    #     print(f"Axis      : {axis}")
    #     print(f"Slice No  : {slice_idx}")
    #     print(f"PIL Size  : {pil_size} (W, H)")
    #     print()

if __name__ == "__main__":
    test_load_cellmap_2d()