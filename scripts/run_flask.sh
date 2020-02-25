
echo "RUN FLASK SCRIPT FOR ENV:"
echo $ENV

# Run development server
flask run --host=0.0.0.0
export FLASK_APP="app.py"
