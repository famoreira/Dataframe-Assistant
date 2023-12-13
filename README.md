# Dataframe-Assistant
Just a simple chat bot that you can use to chat with a dataframe from a CSV save on google drive. I got the original idea from https://github.com/abidsaudagar/talk_with_quran, but no PDF

I wanted to do this so it would load on miniconda, once you load it up, you should be able to use the below to get it to work. Make sure to get a ChatGPT API and replace the value in the .env file.

	git clone https://github.com/famoreira/Dataframe-Assistant dfGPT
	cd dfGPT
	conda create -n dfGPT python=3.11.5 jupyter=1.0.0
	conda activate dfGPT
	pip install -r requirements.txt
	jupyter notebook

#I use the terminal in jupyter, you dont have to, but make sure to run the application with the below:

	streamlit run app.py
