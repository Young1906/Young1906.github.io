dev: 
	hugo server --buildDrafts
build:
	rm -r docs; hugo; cp -r public docs
