import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
vader_analyzer=SentimentIntensityAnalyzer()
text1="I purchased headphones online. i am very happy with the product"
print(vader_analyzer.polarity_scores(text1))
result1=vader_analyzer.polarity_scores(text1)
print("the sentance is rated as",result1['pos']*100,"% Positive")
print("the sentance is rated as",result1['neg']*100,"% Negative")
print("the sentance is rated as",result1['neu']*100,"% Neutral")

if result1['compound']>=0.05:
	print("overall rating for sentance is Positive")
elif result1['compound']<=0.05:
	print("overall rating for sentance is Negative")
else :
	print("overall rating for sentance is neutral")
	
