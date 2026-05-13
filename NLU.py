import joblib
import json
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process,fuzz

vectorizer=joblib.load(r"C:\projects\2nd_Semester_Project\data&models\tfidf_vectorizer.pkl")
SVM_model=joblib.load(r"C:\projects\2nd_Semester_Project\data&models\intent_classifier_svm.pkl")
with open(r"C:\projects\2nd_Semester_Project\data&models\Label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
def clean_text(text):
    table = str.maketrans('', '', 'ًٌٍَُِّْ')
    text = text.translate(table)
    text = text.replace('ى', 'ي').replace('ة', 'ه')
    text = re.sub(r'[أإآا]', 'ا', text)
    text = text.replace('ـ', '')
    text = re.sub(r'(\S)\1{2,}', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation + '؟،؛'))
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

Drivers_db = {
            # ريد بول (Red Bull)
            "ماكس فيرستابن": "max_verstappen",
            "ماكس فرستابن": "max_verstappen",
            "فيرستابن": "max_verstappen",
            "ماكس": "max_verstappen",
            "إيزاك هادجار": "isack_hadjar",
            "ايزاك هادجار": "isack_hadjar",

            # فيراري (Ferrari)
            "لويس هاميلتون": "lewis_hamilton",
            "لويس هاملتون": "lewis_hamilton",
            "هاميلتون": "lewis_hamilton",
            "لويس": "lewis_hamilton",
            "شارل لوكلير": "charles_leclerc",
            "شارل لوكليرك": "charles_leclerc",
            "لوكلير": "charles_leclerc",

            # مكلارين (McLaren)
            "لاندو نوريس": "lando_norris",
            "نوريس": "lando_norris",
            "لاندو": "lando_norris",
            "أوسكار بياستري": "oscar_piastri",
            "اوسكار بياستري": "oscar_piastri",
            "بياستري": "oscar_piastri",

            # مرسيدس (Mercedes)
            "جورج راسل": "george_russell",
            "راسل": "george_russell",
            "جورج": "george_russell",
            "كيمي أنتونيلي": "kimi_antonelli",
            "أنتونيلي": "kimi_antonelli",
            "اندريا كيمي انتونيلي": "kimi_antonelli",

            # أستون مارتن (Aston Martin)
            "فرناندو ألونسو": "fernando_alonso",
            "فرناندو الونسو": "fernando_alonso",
            "ألونسو": "fernando_alonso",
            "لانس سترول": "lance_stroll",
            "سترول": "lance_stroll",

            # ويليامز (Williams)
            "كارلوس ساينز": "carlos_sainz",
            "ساينز": "carlos_sainz",
            "أليكس ألبون": "alex_albon",
            "اليكس البون": "alex_albon",
            "ألبون": "alex_albon",

            # ألبين (Alpine)
            "بيير غاسلي": "pierre_gasly",
            "غاسلي": "pierre_gasly",
            "فرانكو كولابينتو": "franco_colapinto",
            "كولابينتو": "franco_colapinto",

            # هاس (Haas)
            "إستيبان أوكون": "esteban_ocon",
            "استيبان اوكون": "esteban_ocon",
            "أوكون": "esteban_ocon",
            "أوليفر بيرمان": "oliver_bearman",
            "اوليفر بيرمان": "oliver_bearman",
            "بيرمان": "oliver_bearman",

            # آر بي (Racing Bulls)
            "ليام لوسون": "liam_lawson",
            "لوسون": "liam_lawson",
            "أرفيد ليندبلاد": "arvid_lindblad",
            "ليندبلاد": "arvid_lindblad",

            "نيكو هولكنبرغ": "nico_hulkenberg",
            "هولكنبرغ": "nico_hulkenberg",
            "غابرييل بورتوليتو": "gabriel_bortoleto",
            "بورتوليتو": "gabriel_bortoleto",
            
            "سيرجيو بيريز": "sergio_perez",
            "سيرجيو بيرز": "sergio_perez",
            "بيريز": "sergio_perez",
            "تشيكو بيريز": "sergio_perez",
            "فالتيري بوتاس": "valtteri_bottas","بوتاس": "valtteri_bottas",
        }
Circuits_db = {
            "موناكو": "monaco",
            "سيلفرستون": "silverstone",
            "بريطانيا": "silverstone",
            "البحرين": "bahrain",
            "السعودية": "saudi_arabia",
            "جدة": "saudi_arabia",
            "أستراليا": "australia",
            "استراليا": "australia",
            "ملبورن": "australia",
            "اليابان": "japan",
            "سوزوكا": "japan",
            "الصين": "china",
            "شنغهاي": "china",
            "ميامي": "miami",
            "إيمولا": "imola",
            "ايمولا": "imola",
            "كندا": "canada",
            "مونتريال": "canada",
            "إسبانيا": "spain",
            "اسبانيا": "spain",
            "برشلونة": "spain",
            "النمسا": "austria",
            "المجر": "hungary",
            "بودابست": "hungary",
            "بلجيكا": "belgium",
            "سبا": "belgium",
            "هولندا": "netherlands",
            "زاندفورت": "netherlands",
            "إيطاليا": "italy",
            "ايطاليا": "italy",
            "مونزا": "italy",
            "أذربيجان": "azerbaijan",
            "اذربيجان": "azerbaijan",
            "باكو": "azerbaijan",
            "سنغافورة": "singapore",
            "أمريكا": "usa",
            "امريكا": "usa",
            "أوستن": "usa",
            "المكسيك": "mexico",
            "البرازيل": "brazil",
            "إنترلاغوس": "brazil",
            "لاس فيغاس": "las_vegas",
            "فيغاس": "las_vegas",
            "قطر": "qatar",
            "لوسيل": "qatar",
            "أبوظبي": "abu_dhabi",
            "أبو ظبي": "abu_dhabi",
            "ياس مارينا": "abu_dhabi",
        }
Arabic_num = {
            "سباق": "1","واحد": "1",
            "اثنين": "2","اتنين": "2","سباقين": "2",
            "ثلاثة": "3","تلاثة": "3","ثلاث":"3","اربعة": "4",'الاربع':"4",
            "أربعة": "4","خمسة": "5","الخمس":"5","ستة": "6","الست":"6",
            "سبعة": "7","ثمانية": "8","تمانية": "8",
            "تسعة": "9","عشرة": "10","عشر": "10"
        }
Time_direction = {
            # Future
            "الجاي": "future","الجاية": "future",
            "بكرا": "future","بكرة": "future",
            "القادم": "future","القادمة": "future",
            
            # Past
            "الماضي": "past",
            "الماضية": "past","الفات": "past",
            "اللي فات": "past","السابق": "past",
            "السابقة": "past", "اليوم": "present",
            "الآن": "present", "الان": "present",
            "هلق": "present","هسا": "present"
        }
Teams_db = {
            # Red Bull
            "ريدبول": "red_bull",
            "الريدبول": "red_bull",
            "ريد بول": "red_bull",
            
            # Ferrari
            "فيراري": "ferrari",
            "الفيراري": "ferrari",
            
            # Mercedes
            "ميرسيدس": "mercedes",
            "مرسيدس": "mercedes",
            "المرسيدس": "mercedes",
            
            # McLaren
            "مكلارين": "mclaren",
            "ماكلارين": "mclaren",
            "المكلارين": "mclaren",
            
            # Aston Martin
            "أستون مارتن": "aston_martin",
            "استون مارتن": "aston_martin",
            
            # Williams
            "ويليامز": "williams",
            "وليامز": "williams",
            
            # Alpine
            "ألبين": "alpine",
            "البين": "alpine",
            
            # Haas
            "هاس": "haas",
            
            # Racing Bulls (RB)
            "آر بي": "racing_bulls",
            "ار بي": "racing_bulls",
            "ريسينغ بولز": "racing_bulls",
            
            # Audi / Sauber
            "أودي": "audi",
            "اودي": "audi",
            "ساوبر": "audi",
            
            # Cadillac
            "كاديلاك": "cadillac",
        }
def MatchingAndFuzzMatching(Text):
        keys={"Circuits":Circuits_db,"Teams": Teams_db,"Drivers": Drivers_db}
        fixed_keys={"Time":Time_direction,"Number":Arabic_num}
        english_ID={}
    
        #match each word to find if its in time or number db and add them to the extarcted Text in english
        for word in Text.split():
            for name,db in fixed_keys.items():
                    if word in db:
                        english_ID[name]=db[word]
    
        #match the whole sentence using rapid fuzz technique which find the highest matched db so we can find the right command to trigger 
        for key,db in keys.items():
            results=process.extract(Text,db.keys(),scorer=fuzz.partial_ratio,limit=2)
            matched = {
            db[r[0]] for r in results if r[1] > 80
            }
            if matched:
                matched=list(matched)
                english_ID[key] = matched if len(matched) > 1 else matched[0]
        return english_ID

def process_query(Text) :
        cleaned = clean_text(Text)
        vectorized = vectorizer.transform([cleaned])
        intent = SVM_model.predict(vectorized)[0]
        
        # Extract entities from original Text (not cleaned — preserve numerals)
        entities = MatchingAndFuzzMatching(Text)
        
        return {
            "intent": intent,
            "entities": entities,
            "raw_Text": Text
        }
