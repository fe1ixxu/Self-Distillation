import argparse
import os



def iwslt(args):
    # langs = ["ar", "fa", "de", "es", "it", "nl", "pl", "pt", "ro", "sl", "tr", "ru", "he", "zh"]
    langs = "de,es,it,nl,pl,ar,fa,he".split(",")
    bleus = []
    
    for lg in langs:
        if args.nen:
            bleu_file = f"test.en-{lg}.{lg}.bleu"
        else:
            bleu_file = f"test.{lg}-en.en.bleu"
        bleu_file = os.path.join(args.path, bleu_file)
        if os.path.exists(bleu_file):
            print(f"{lg}")
            with open(bleu_file) as f:
                bleu = float(f.readlines()[0].strip())
            
            bleus.append(bleu)
            print(f"Bleu for {lg} is {round(bleu, 2)}")
        else:
            print(f"No Bleu for {lg}")


    print(f"Mean value of BLEU is {sum(bleus)/len(bleus)}")
    print("\t".join([str(b) for b in bleus]))

def opus_test(args):
    langs = ["de", "es", "it", "nl", "pl", "ar", "fa", "he"]
    bleus = []
    for lg in langs:
        if args.nen:
            bleu_file = f"test.en-{lg}.{lg}.bleu" 
        else:
            bleu_file = f"test.en-{lg}.en.bleu"
        bleu_file = os.path.join(args.path, bleu_file)
        if os.path.exists(bleu_file):
            print(f"{lg}")
            with open(bleu_file) as f:
                bleu = float(f.readlines()[0].strip())
            
            bleus.append(bleu)
            print(f"Bleu for {lg} is {round(bleu, 2)}")
        else:
            print(f"No Bleu for {lg}")


    print(f"Mean value of BLEU is {sum(bleus)/len(bleus)}")
    print("\t".join([str(b) for b in bleus]))


def wmt(args):
    langs = "hr sr mk et hu jv id ms tl".split(" ")
    bleus = []
    for lg in langs:
        if args.nen:
            bleu_file = f"test.en-{lg}.{lg}.bleu" 
        else:
            bleu_file = f"test.en-{lg}.en.bleu"
        bleu_file = os.path.join(args.path, bleu_file)
        if os.path.exists(bleu_file):
            print(f"{lg}")
            with open(bleu_file) as f:
                bleu = float(f.readlines()[0].strip())
            
            bleus.append(bleu)
            print(f"Bleu for {lg} is {round(bleu, 2)}")
        else:
            print(f"No Bleu for {lg}")


    print(f"Mean value of BLEU is {sum(bleus)/len(bleus)}")
    print("\t".join([str(b) for b in bleus]))




def opus(args):
    langs = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'eo', 'es', 'et',
            'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'id', 'ig', 'is', 'it', 'ja',
            'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'li', 'lt', 'lv', 'mg', 'mk', 'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'ne',
            'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'rw', 'se', 'sh', 'si', 'sk', 'sl', 'sq', 'sr',
            'sv', 'ta', 'te', 'tg', 'th', 'tk', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'wa', 'xh', 'yi', 'zh', 'zu']

    high_langs = {'nl', 'ca', 'fi', 'mk', 'da', 'cs', 'bg', 'ro', 'is', 'th', 'he', 'uk', 'lv', 'pl', 'pt', 'hu', 'de', 'lt',
                  'si', 'ms', 'sv', 'tr', 'ko', 'sq', 'el', 'fa', 'es', 'zh', 'bs', 'ar', 'eu', 'fr', 'bn', 'it', 'sk', 'sr',
                  'et', 'vi', 'mt', 'no', 'sl', 'id', 'ja', 'ru', 'hr'}
    med_langs = {'ga', 'af', 'tg', 'gu', 'km', 'sh', 'hi', 'rw', 'nb', 'wa', 'uz', 'ka', 'ml', 'ur', 'gl', 'br', 'cy', 'ku',
                 'ne', 'pa', 'mg', 'as', 'eo', 'xh', 'nn', 'ta', 'az', 'tt'}
    low_langs = {'fy', 'mr', 'tk', 'kn', 'li', 'yi', 'my', 'zu', 'ug', 'or', 'se', 'am', 'oc', 'ig', 'ha', 'ky', 'te', 'be',
                 'kk', 'gd', 'ps'}
                 
    high_bleu, med_bleu, low_bleu, all_bleu = [], [], [], []
    for lg in langs:
        if args.nen:
            bleu_file = f"test.en-{lg}.{lg}.bleu" 
        else:
            bleu_file = f"test.en-{lg}.en.bleu"
        with open(os.path.join(args.path, bleu_file)) as f:
            bleu = float(f.readlines()[0].strip())

        all_bleu.append(bleu)

        if lg in high_langs:
            high_bleu.append(bleu)
        elif lg in med_langs:
            med_bleu.append(bleu)
        elif lg in low_langs:
            low_bleu.append(bleu)
        else:
            raise ValueError("No such language")

    print("mean of bleu of high langs is {}, with number of {} langs". format(sum(high_bleu)/len(high_bleu), len(high_bleu)))
    print("mean of bleu of med langs is {}, with number of {} langs". format(sum(med_bleu)/len(med_bleu), len(med_bleu)))
    print("mean of bleu of low langs is {}, with number of {} langs". format(sum(low_bleu)/len(low_bleu), len(low_bleu)))
    print("mean of bleu of all langs is {}, with number of {} langs". format(sum(all_bleu)/len(all_bleu), len(all_bleu)))


def get_lgs(args):
    picked = ["fr", "es", "de", "pl", "cs", "mk", "bg", "uk", "be", "ru", "lv", "lt", "et", "fi", "hi", "mr", "ne", "kk", "ky", "ar", "ps", "fa", "zu", "xh", "id", "ms"]
    langs = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'eo', 'es', 'et',
            'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'id', 'ig', 'is', 'it', 'ja',
            'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'li', 'lt', 'lv', 'mg', 'mk', 'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'ne',
            'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'rw', 'se', 'sh', 'si', 'sk', 'sl', 'sq', 'sr',
            'sv', 'ta', 'te', 'tg', 'th', 'tk', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'wa', 'xh', 'yi', 'zh', 'zu']

    high_langs = {'nl', 'ca', 'fi', 'mk', 'da', 'cs', 'bg', 'ro', 'is', 'th', 'he', 'uk', 'lv', 'pl', 'pt', 'hu', 'de', 'lt',
                  'si', 'ms', 'sv', 'tr', 'ko', 'sq', 'el', 'fa', 'es', 'zh', 'bs', 'ar', 'eu', 'fr', 'bn', 'it', 'sk', 'sr',
                  'et', 'vi', 'mt', 'no', 'sl', 'id', 'ja', 'ru', 'hr'}
    med_langs = {'ga', 'af', 'tg', 'gu', 'km', 'sh', 'hi', 'rw', 'nb', 'wa', 'uz', 'ka', 'ml', 'ur', 'gl', 'br', 'cy', 'ku',
                 'ne', 'pa', 'mg', 'as', 'eo', 'xh', 'nn', 'ta', 'az', 'tt'}
    low_langs = {'fy', 'mr', 'tk', 'kn', 'li', 'yi', 'my', 'zu', 'ug', 'or', 'se', 'am', 'oc', 'ig', 'ha', 'ky', 'te', 'be',
                 'kk', 'gd', 'ps'}

    l, m, h = [], [], []
    for lg in picked:
        if lg in high_langs:
            h.append(lg)
        elif lg in med_langs:
            m.append(lg)
        elif lg in low_langs:
            l.append(lg)
    print("high langs {} is with number of {} langs".format(h, len(h)))
    print("med langs {} is with number of {} langs".format(m, len(m)))
    print("low langs {} is with number of {} langs".format(l, len(l)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--nen', action="store_true")
    parser.add_argument('--data', type=str, choices=["iwslt", "opus", "opus_test", "wmt"])
    args = parser.parse_args()

    if args.data == "iwslt":
        iwslt(args)
    elif args.data == "opus":
        opus(args)
    elif args.data == "opus_test":
        opus_test(args)
    elif args.data == "wmt":
        wmt(args)
    else:
        raise ValueError("Not such dataset")