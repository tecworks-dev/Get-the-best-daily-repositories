import sys, os
from elftools.elf.elffile import ELFFile

def extract_text_section(elf_path, output_path):
    if not os.path.exists(elf_path):
        print("[!] File doesnt exist")
        sys.exit(1)
        
    with open(elf_path, 'rb') as elf_file:
        elf = ELFFile(elf_file)
        
        text_section = elf.get_section_by_name('.text')
        if not text_section:
            print("No .text section found in the ELF file.")
            return

        text_data = text_section.data() 
        with open(output_path, 'wb') as output_file:
            output_file.write(text_data)
        
        print(f".text section extracted successfully to {output_path}")

# Example usage
extract_text_section(sys.argv[1], 'payload.bin')
