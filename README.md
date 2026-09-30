# Patient Appointment No-Show Prediction

**Applied Analytics & AI Mini Project**

**Abdolhossein (Benjamin) Ayoubi**
Master's Degree Programme in Smart Industry
Metropolia University of Applied Sciences, 2026

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem and Objective](#1-problem-and-objective)
3. [Data and Cleaning](#2-data-and-cleaning)
4. [What I Found in the Data](#3-what-i-found-in-the-data)
5. [Modeling](#4-modeling)
6. [Interpreting the Result](#5-interpreting-the-result)
7. [Responsible AI and Practical Use](#6-responsible-ai-and-practical-use)
8. [Limitations and Next Steps](#7-limitations-and-next-steps)
9. [AI Use Note](#8-ai-use-note)
10. [References](#references)

---

## Executive Summary

The practical question: **can appointment data help spot patients who are more likely to miss a visit?**

I used the public [Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) dataset from Kaggle. It starts with 110,527 appointments; after a small cleaning step, 110,521 records remained.

**Key results**

- **No-show rate:** 20.19% of all appointments.
- **Strongest early signal:** waiting time. The no-show rate was **4.65%** for same-day appointments but reached **34.15%** when the appointment was booked 31 to 60 days ahead.
- **Models tested:** a dummy baseline, Logistic Regression, Decision Tree, and Random Forest, using a 70/30 split by `PatientId` so the same patient never appears in both sets.
- **Best model:** Random Forest, with the strongest no-show F1 score.

| Precision | Recall | F1    | ROC-AUC |
|----------:|-------:|------:|--------:|
| 0.304     | 0.845  | 0.447 | 0.728   |

The model catches many missed appointments, but it also creates many false alarms. I would use it for **reminders or scheduling support, not for automatic decisions about patients.**

## 1. Problem and Objective

Missed appointments waste time and make scheduling harder. The goal was to find patterns linked to no-shows, then test whether a basic classification workflow could flag higher-risk appointments.

> **Research question:** Can appointment-level data identify patients with a higher probability of missing a scheduled appointment?

I treated `No-show = Yes` as the positive class. Since only about one appointment in five is a no-show, accuracy alone can be misleading. I focused on precision, recall, F1-score, ROC-AUC, and the confusion matrix.

## 2. Data and Cleaning

The dataset includes patient characteristics, scheduling dates, appointment dates, SMS status, neighbourhood, and the final attendance outcome. Before modeling, I checked the raw CSV for missing values, duplicate rows, impossible ages, and negative waiting times.

| Check                 | Result  |
|-----------------------|--------:|
| Original rows         | 110,527 |
| Rows after cleaning   | 110,521 |
| Missing values        | 0       |
| Exact duplicate rows  | 0       |
| Age below 0           | 1       |
| Negative waiting days | 5       |

- Removed **1 row** with a negative age and **5 rows** with negative waiting time.
- Kept age 0, since it can represent infants.
- Kept `PatientId` out of the model and used it only to make the train/test split.

## 3. What I Found in the Data

### Waiting time

Waiting time stood out almost immediately. Same-day appointments had a 4.65% no-show rate. The rate increased as the wait became longer and reached 34.15% in the 31 to 60 day group. It dropped again for 61+ days, so the relationship is strong but not perfectly linear.

![Figure 1: No-show rate by waiting-time group](images/fig1-no-show-by-waiting-time.png)

*Figure 1. No-show rate by waiting-time group.*

### Age and gender

- The **18 to 29** group had the highest no-show rate at **24.64%**, with ages **6 to 17** close behind at **24.36%**.
- Gender barely changed the result: **20.31%** for women and **19.96%** for men.

### SMS reminders

The SMS result looked surprising at first: patients who received an SMS had a **27.57%** no-show rate, compared with **16.70%** for those who did not.

I do not read that as proof that SMS reminders cause no-shows. The reminders were probably not assigned randomly, so higher-risk or later appointments may have received them more often.

## 4. Modeling

I kept the modeling stage straightforward.

**Features and exclusions**

- Created time-based features: waiting days, scheduled hour, appointment weekday, scheduled weekday, and appointment month.
- Removed `AppointmentID` because it is only an identifier.
- Left `Neighbourhood` out of the final model because location can act as a proxy for social or geographic factors.

**Train/test split**

I split the data 70% / 30% by `PatientId`, giving **77,565 training rows** and **32,956 test rows**. This matters because some patients appear more than once. A normal row-level split could put the same person in both sets and make the test score look better than it really is.

**Models compared**

1. Majority-class dummy baseline
2. Balanced Logistic Regression
3. Balanced Decision Tree
4. Balanced Random Forest

I kept the tree settings limited on purpose. I wanted models I could explain, not models that simply memorize the training data.

| Model               | Accuracy | Precision | Recall |    F1 | ROC-AUC |
|---------------------|---------:|----------:|-------:|------:|--------:|
| Dummy baseline      |    0.796 |     0.000 |  0.000 | 0.000 |   0.500 |
| Logistic Regression |    0.663 |     0.320 |  0.574 | 0.410 |   0.669 |
| Decision Tree       |    0.573 |     0.303 |  0.835 | 0.444 |   0.725 |
| **Random Forest**   |    0.573 |     0.304 |  0.845 | 0.447 |   0.728 |

*Table 1. Test-set results. No-show is the positive class.*

![Figure 2: Precision, recall, F1 and ROC-AUC for the three trained classifiers](images/fig2-model-metrics.png)

*Figure 2. Precision, recall, F1, and ROC-AUC for the three trained classifiers.*

## 5. Interpreting the Result

The dummy baseline reached 0.796 accuracy, which sounds good until you look closer. It predicted every appointment as a show, so it found zero no-shows. That is exactly why accuracy was not the main measure.

Random Forest gave the best F1 score for the no-show class. Its recall was 0.845, so it found about **84.5%** of actual no-shows in the test set. Precision was only 0.304, so many people flagged by the model would still attend. For an extra reminder, that trade-off is acceptable. For anything that affects access to care, it is not.

**Confusion matrix at the default 0.50 threshold**

|                     | Predicted no-show | Predicted show |
|---------------------|------------------:|---------------:|
| **Actual no-show**  |   5,692 (TP)      |   1,043 (FN)   |
| **Actual show**     |  13,020 (FP)      |  13,201 (TN)   |

### Threshold check

Lower thresholds caught more no-shows but flagged far more appointments. A higher threshold reduced false alarms, but recall dropped quickly. I would not choose a threshold from model scores alone: the right setting depends on how costly a missed appointment is compared with sending an unnecessary reminder.

| Threshold | Precision | Recall |    F1 | Flagged |
|----------:|----------:|-------:|------:|--------:|
|      0.30 |     0.287 |  0.932 | 0.439 |  66.25% |
|      0.40 |     0.288 |  0.921 | 0.439 |  65.27% |
|  **0.50** |     0.304 |  0.845 | 0.447 |  56.78% |
|      0.60 |     0.353 |  0.521 | 0.421 |  30.22% |

*Table 2. Random Forest threshold trade-off on the test set.*

### Feature importance

![Figure 3: Top Random Forest feature importances](images/fig3-feature-importance.png)

*Figure 3. Top Random Forest feature importances.*

Waiting days dominated the Random Forest feature importance. Age, SMS status, and scheduled hour also contributed. I treat this as a clue about what the model uses, not as proof of cause and effect.

## 6. Responsible AI and Practical Use

I would keep the real-world use narrow. The model could help a scheduling team send an extra reminder, ask for confirmation, or offer an easier rescheduling option. These are low-risk actions, and people still make the final decision.

- **Fairness:** the model did not perform equally across every age group. Recall was 0.942 for ages 0 to 5 and 30 to 44, but only **0.593** for patients aged 60+. Gender recall was closer, at 0.856 for women and 0.824 for men. I would treat the age gap as a warning sign and monitor it before any real use.
- **Privacy:** I used a public dataset here. Real healthcare data would need strict access control, data minimization, and suitable anonymization or pseudonymization.
- **Explainability:** staff should understand why the model raises a flag. Waiting time is easy to explain. An opaque identifier is not.
- **Human oversight:** the prediction should support scheduling work. It should never cancel an appointment, lower a patient's priority, or reduce access automatically.

## 7. Limitations and Next Steps

**Limitations**

- This is one historical public dataset. I would not assume the same percentages apply to a Finnish hospital or to patients today.
- The analysis is observational, so it shows associations rather than causes.
- The model would need external validation before real use.

**Next steps**

1. Test the model on newer local data.
2. Choose the threshold using the real cost of reminders and missed appointments.
3. Track performance over time.
4. Repeat the subgroup checks after deployment.

## 8. AI Use Note

I carried out the main parts of the project myself, including data cleaning, feature preparation, model selection, evaluation, and interpretation of the results. I used AI only as a supporting tool for occasional coding help and language editing. I reviewed the code, reran the notebook, checked the outputs, and made the final decisions about the methods, results, and conclusions.

## References

- Kaggle. *Medical Appointment No Shows* dataset. <https://www.kaggle.com/datasets/joniarroba/noshowappointments>
- Metropolia University of Applied Sciences. *Applied Analytics & AI* course materials, 2026.
