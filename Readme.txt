Readme
1. In the same directory that contains the 'test_queries.py' and 'build_index.py' files, create a directory called 'wikis' and paste the relevant wiki files to be used as the corpus for the system. For the results in the report, the wikis used are "AA - wiki_10 to wiki_29 (20 files)".
2. Run the 'build_index.py' file once. This will create two pickle files, namely 'index.pickle' and 'modified_index.pickle' in the same directory as of 'build_index.py'. The index has been created. If you change the corpus, this file has to be run again to change the pickle files.
	Command: 	python3 build_index.py
3. Run the test_queries.py file to test the queries. This file can be run multiple times for different queries, as long as the corpus is fixed.
	Command: 	python3 test_queries.py
   Input query, path of index(default or new), whether to consider improvements or not and value of k needs to supplied as and when asked in program.
