python -m venv .tot
source .tot/bin/activate
pip install --upgrade pip 
pip install -r requirements.txt
pip install -e . 

# Adjust environment variables
echo "export HUGGINGFACE_TOKEN=<HUGGINGFACE_TOKEN>" >> .tot/bin/activate
echo "export OPENAI_API_KEY=<OPENAI_API_KEY>" >> .tot/bin/activate
echo "export GEMINI_API_KEY=<GEMINI_API_KEY>" >> .tot/bin/activate