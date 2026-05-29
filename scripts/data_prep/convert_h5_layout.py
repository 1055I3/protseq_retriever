import h5py
import os
import tqdm

def convert_h5_to_compatible(source_path, target_path):
    print(f"Starting conversion: {source_path} -> {target_path}")
    
    if not os.path.exists(source_path):
        print(f"Error: Source file {source_path} not found.")
        return

    # Open source with 'latest' to ensure we can read modern layout messages
    # Create target with 'earliest' to maximize backward compatibility
    try:
        with h5py.File(source_path, 'r', libver='latest') as src, \
             h5py.File(target_path, 'w', libver='earliest') as dst:
            
            keys = list(src.keys())
            print(f"Found {len(keys)} datasets to copy...")
            
            for key in tqdm.tqdm(keys, desc="Converting"):
                if isinstance(src[key], h5py.Dataset):
                    data = src[key][:]
                    # Create dataset with standard chunked layout and no compression
                    # Matching per-protein.h5 style (universally readable)
                    dst.create_dataset(
                        key, 
                        data=data, 
                        chunks=True,      # Standard chunked layout
                        compression=None  # No compression for max compatibility
                    )
        
        print(f"\nSuccess! Compatible HDF5 created at: {target_path}")
        
    except Exception as e:
        print(f"\nConversion failed: {str(e)}")

if __name__ == "__main__":
    src_file = 'bioseq_retriever/data/per-gene.h5'
    backup_file = 'bioseq_retriever/data/per-gene.h5.compact'
    
    # We will write to a temp file and then rename
    temp_target = 'bioseq_retriever/data/per-gene_compatible.h5'
    
    convert_h5_to_compatible(src_file, temp_target)
    
    print("\nTo use the new file, run:")
    print(f"  mv {src_file} {backup_file}")
    print(f"  mv {temp_target} {src_file}")
