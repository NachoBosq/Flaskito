from flask import Flask, request, jsonify
import cohere

co = cohere.Client('pzoFCrIVcNOxfizmKGkx5IPSG1LUfBoE0BNmS1Ic')

app = Flask(__name__)
@app.route("/", methods=["GET"])
def welcome():
        return "Hello World!"

@app.route("/bad_language_detector", methods=["GET"])
def detector():
    text = request.args.get("text")
    if text:
        prompt = f"""
I am going to give you a text. You need to check if there is a bad language in the text.
In case there is, return a 1. Otherwise, if the text is fine return a 0.
IMPORTANT: Don't give any details or metadata. Just 0 or 1 depending on the content.
TEXT: {text}
RESPONSE: (0 OR 1)
"""
        response = co.generate(
            model='medium',  # specify the model to use
            prompt=prompt,
            max_tokens=1,  # specify the maximum number of tokens to generate
            temperature=0.01,  # set the temperature
            k=0,  # top-k sampling parameter
            p=0.75,  # top-p sampling parameter
            stop_sequences=[],  # list of stop sequences
            return_likelihoods="NONE"  # specify likelihood return option
        )
        return response.generations[0].text
    else:
        return "Cosetes"
    
if "__main__" == __name__:
        app.run(debug=True)