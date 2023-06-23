dev: 
	hugo server --buildDrafts
build:
	hugo; cp -r public docs
