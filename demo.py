from svo_extraction.subject_verb_object_extract import invertSentence
import random
import time

# import en_core_web_sm

start = time.time()
shuff = (invertSentence("The americans went to the moon."))
end = time.time()
print (shuff, end-start)
