from scripts import get_metadata, sample_data_script, swap_script

def main():
    # Run each phase script
    try:
        print("Starting get_metadata...")
        get_metadata.main()
        print("get_metadata completed.\n")
        
        print("Starting sample_data_script...")
        sample_data_script.main()
        print("sample_data_script completed.\n")
        
        print("Starting swap_script...")
        swap_script.main()
        print("swap_script completed.\n")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
