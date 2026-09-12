# Data

The main dataset is the Kaggle **Customer Invoices Dataset — Payment Date Prediction on Open Invoices** by Pradumn Mishra:
https://www.kaggle.com/datasets/pradumn203/payment-date-prediction-for-invoices-dataset

License shown on Kaggle: CC BY-NC 4.0. Download with:
`python scripts/download_data.py`

For local smoke testing without external access:
`python scripts/generate_demo.py`
`python -m src.train --source demo`

Do not commit the downloaded Kaggle files into GitHub unless their license permits your intended redistribution.
