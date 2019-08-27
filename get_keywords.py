import helpers
import pickle

with open('./models/tfidf_transformer.pkl', 'rb') as fp:
    tfidf_transformer = pickle.load(fp)
with open('./models/count_vectorizer.pkl', 'rb') as fp:
    count_vectorizer = pickle.load(fp)
feature_names = count_vectorizer.get_feature_names()

doc = """
Imagine what you could do here. At Apple, extraordinary ideas have a way of becoming great products, services, and customer experiences very quickly. Bring passion and dedication to your job and there's no telling what you could accomplish. Apple’s University Recruiting team is looking for a highly motivated, engineering students with a strong background in Back-End Engineering, Core OS, and Web Development to join its team of highly skilled software engineers. Our software engineers are the brains behind some of the industry’s biggest breakthroughs. OS X, Siri, Apple Maps, and iCloud — not to mention the system-level software for iPhone and Apple TV — all started here. These teams are on the front line of our constant charge toward innovation. We are actively seeking enthusiastic interns who can work full-time for a minimum of 12-weeks either for our Fall 2019, Winter 2020, Spring 2020, or Summer 2020 sessions.
Key Qualifications
Strong object-oriented design skills, coupled with a deep knowledge of data structures and algorithms
Proficiency in one or more of the following developer skills: Java, C/C++, PHP, Python, Ruby, Unix, MySQL, Clojure, Scala, Java Script, CSS, HTML5
Experience in advanced Big Data methodologies such as Data Modeling, Validation, Processing, Hadoop, MapReduce, Mongo, Pig
Experience with web frameworks such as AngularJS, NodeJS, SproutCore
Demonstrable experience in application development in Objective-C for OS X or iOS a plus
Client-Server protocol & API design Skills
Able to craft multi-functional requirements and translate them into practical engineering tasks
A fundamental knowledge of embedded processors, with in-depth knowledge of real time operating system concepts.
Excellent debugging and critical thinking skills
Excellent analytical and problem-solving skills
Ability to work in a fast paced, team-based environment
Strong verbal and written communication skills and social skills
Description
Backend Development - You are responsible for making the features that our users love (like Siri) work by presenting data to the user-facing applications. Backend development opportunities are available for students in the following areas: Siri, iCloud, Apple Maps, Core OS, OS X, Frameworks and Applications, Interactive Media Group, Audio/Video Software Integration and Localization, Advanced Computation, iLife, iWorks, Aperture, Pro Apps, iTunes, Security, Site Reliability Engineering (SRE) and Platform Infrastructure Engineering (PIE) Core OS - You want to be at the heart of Apple's Software organization. The Core OS team is responsible for the design and development of core technologies that are deployed across all Apple product areas including the iPhone, iPad, Watch, MacBook, iMac, Apple TV, and audio accessories. (Yes, that's pretty much everything.) Web Development - You will help build web-based tools and applications to enhance our products and do more for our customers. Our developers are responsible for shaping the direction of our products by considering the architecture, performance, testing, design, and implementation. And of course we look for engineers that use our products. Engineers here work on both UI level and lower-level implementation details. The successful intern candidate will be amenable to working in a dynamic, collaborative environment. The person filling this position must be a hands-on, enthusiastic, self-motivated developer with strong initiative and desire to succeed in a challenging environment. You will have a real passion for extraordinary user experiences and an eye for details. Those applying for the Web Development intern position should include a link to a web portfolio. Opportunities available in Cupertino and Bay Area, Seattle, Los Angeles, and Austin Apple is an Equal Employment Opportunity Employer that is committed to inclusion and diversity. We also take affirmative action to offer employment and advancement opportunities to all applicants, including minorities, women, protected veterans, and individuals with disabilities.
Education & Experience
BS/MS/PhD program in Computer Science, Electrical Engineering, Computer Engineering, Data Science, Design, or related fields. You are additionally required to return to school after the internship to continue or complete your education, or an internship needs to be required for graduation from your school. You also qualify if accepted into a graduate program after graduation.
"""
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(count_vectorizer.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items = helpers.sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords = helpers.extract_topn_from_vector(feature_names,sorted_items, 100)