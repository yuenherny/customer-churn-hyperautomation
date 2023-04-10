# Maxis Interview Case Study: Machine Learning and RPA for Customer Churn Prediction

## Applied Role / Position
Data Scientist (Internal Audit)

## Details about the Case Study
Please refer to the attached document `CA Specialist Case Study 2.docx` for more info.

## Software Tools
1. Data science and ML: Python and VSCode
2. Automation: UiPath Studio Community Edition 2021.10.5

## Automation Workflow
The automation workflow for the case study is executed using UiPath bot.

### Pre-inference
1. Open browser
2. At browser, go to an Onedrive link
3. Click to download data
4. Extract/Move file from Downloads to selected folder

### Inference
1. Ask user for filename via dialog box
2. Save the filename as variable
3. Run `python predict.py <FILE_PATH>` in terminal

### Email
1. Get the results file path
2. Use Outlook component to email to yuenhern.yu@gmail.com