from seqeval.metrics import (

classification_report,

f1_score,

precision_score,

recall_score

)



true_labels=[

[
"B-PHONE",
"O",
]

]



pred_labels=[

[
"B-PHONE",
"O"

]

]



print(

classification_report(

true_labels,

pred_labels

)

)



print(

"F1:",

f1_score(

true_labels,

pred_labels

)

)