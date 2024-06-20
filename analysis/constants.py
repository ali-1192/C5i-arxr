glasses_tags = ['google', 'lenovo', 'snap', 'apple', 'vuzix', 'xreal', 'rokid',
       'meta']


headset_tags = ["apple", "dpvr", "google", "hp", "htc", "magic leap", "meta", "microsoft", "oppo", "pico", "samsung", "sony/playstation", "valve"]

category_tags =  ['amazon', 'apple', 'google', 'meta', 'microsoft']



query_tags = {'glasses': glasses_tags, 'headsets': headset_tags, 'category': category_tags}

sentiment_dict = {
    'neutral': 1, 
    'negative': 0,
    'positive':2
}

SAMPLE_SIZE = 5000