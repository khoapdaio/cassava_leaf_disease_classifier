
# Cassava Leaf Disease Classifier 

This is a python source code to predict leaf. With my code, you can run the application to recognition leaf through file png or jpg

## Dataset

[Cassava Leaf Data](https://storage.googleapis.com/emcassavadata/cassavaleafdata.zip)

## Trained model: LeNet
You could find my trained model at `lenet_model_cassava.pt`
## Run Locally

Clone the project

```bash
    git clone https://github.com/khoapdaio/cassava_leaf_disease_classifier.git
```

Go to the project directory

```bash
    cd cassava_leaf_disease_classifier
```

Create new enviroment
```bash
    python3 -m venv env
```

Activate new enviroment
```bash
    source env/bin/activate
```

Install dependencies

```bash
    pip install -r requirements.txt
```

Start app

```bash
    streamlit run app.py
```


## License

[![GitHub license](https://img.shields.io/github/license/khoapdaio/cassava_leaf_disease_classifier)](https://github.com/khoapdaio/cassava_leaf_disease_classifier/blob/main/LICENSE)
