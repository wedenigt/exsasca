import gdown

url = "https://drive.google.com/file/d/1DWr4QPiaOKWRQaw0cWP6Ju-EtBMk6pGG/view?usp=drive_link"
output = "out.sdd"

gdown.download(url, output, quiet=False, fuzzy=True)