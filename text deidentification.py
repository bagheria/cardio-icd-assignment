import pandas as pd
import deduce  # pip install git+https://github.com/vmenger/deduce.git


data = pd.read_excel(r'D:\Github\ICD10 Classification\ICD10 letters age sex_SENSITIVE.xlsx')
# df1 = pd.DataFrame(data)
d = []
for i in range(data.shape[0]):
        text = data['UitgaandeBriefTekst_DOC'][i]
        annotated = deduce.annotate_text(
                text,  # The text to be annotated
                patient_first_names="",  # First names (separated by whitespace)
                patient_initials="",
                patient_surname="",
                patient_given_name="",  # Given name
                names=True,  # Person names, including initials
                locations=True,  # Geographical locations
                institutions=True,
                dates=True,
                ages=True,
                patient_numbers=True,
                phone_numbers=True,
                urls=True,  # Urls and e-mail addresses
                flatten=True  # Debug option
        )
        de_identified = deduce.deidentify_annotations(annotated)
        d.append(de_identified)

df2 = pd.DataFrame(d, columns=['deidentified'])
result = pd.concat([data, df2], axis=1)
result.to_excel("D:\Github\ICD10 Classification\l_anonym.xlsx")
