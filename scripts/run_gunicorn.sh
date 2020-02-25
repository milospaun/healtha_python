#!/bin/bash

#if [[ $ENV != "dev" &&  "$ENV" != "stage" && "$ENV" != "prod" ]]; then
#    echo "Available inputs are: 'dev' 'stage' 'prod'"
#    echo "TO run: bash projectRoot/scripts/build.sh"
#    exit 1
#fi
echo "RUN FLASK SCRIPT FOR ENV:"
echo $ENV

# Run development server
#flask run --host=0.0.0.0
#export FLASK_APP="flaskr:create_app('$ENV')"

# run prodcution server Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "flaskr:create_app('$ENV')"
