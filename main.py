from flask import Flask, request, render_template, redirect, url_for, session

app = Flask(__name__)

@app.route('/')
def hello():
    return 'hello, world!'

if __name__ == '__main__':
    app.run(debug=True) 



