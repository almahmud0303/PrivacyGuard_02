LABELS = [

"O",

"B-PERSON",

"B-EMAIL",

"B-PHONE",

"B-NID",

"B-LOCATION"

]


label2id = {
    label:i
    for i,label in enumerate(LABELS)
}


id2label = {
    i:label
    for i,label in enumerate(LABELS)
}