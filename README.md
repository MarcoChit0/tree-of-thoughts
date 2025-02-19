conda env create -f environment.yml
conda activate tot
pip install --upgrade pip
pip install -e . 

# Adjust environment variables
conda env config vars set HUGGINGFACE_TOKEN=<HUGGINGFACE_TOKEN>
conda env config vars set OPENAI_API_KEY=<OPENAI_API_KEY>
conda env config vars set GEMINI_API_KEY=<GEMINI_API_KEY>
