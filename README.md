# telco-customer-churn-prediction
To predict whether a customer will change telecommunications provider i.e. "churn".

The training dataset contains 4250 samples. Each sample contains 19 features and 1 boolean variable "churn" which indicates the class of the sample. The 19 input features and 1 target variable are:

"state", string. 2-letter code of the US state of customer residence
"account_length", numerical. Number of months the customer has been with the current telco provider
"area_code", string="area_code_AAA" where AAA = 3 digit area code.
"international_plan", (yes/no). The customer has international plan.
"voice_mail_plan", (yes/no). The customer has voice mail plan.
"number_vmail_messages", numerical. Number of voice-mail messages.
"total_day_minutes", numerical. Total minutes of day calls.
"total_day_calls", numerical. Total minutes of day calls.
"total_day_charge", numerical. Total charge of day calls.
"total_eve_minutes", numerical. Total minutes of evening calls.
"total_eve_calls", numerical. Total number of evening calls.
"total_eve_charge", numerical. Total charge of evening calls.
"total_night_minutes", numerical. Total minutes of night calls.
"total_night_calls", numerical. Total number of night calls.
"total_night_charge", numerical. Total charge of night calls.
"total_intl_minutes", numerical. Total minutes of international calls.
"total_intl_calls", numerical. Total number of international calls.
"total_intl_charge", numerical. Total charge of international calls
"number_customer_service_calls", numerical. Number of calls to customer service
"churn", (yes/no). Customer churn - target variable.
