dev: 
	hugo server --buildDrafts
build:
<<<<<<< HEAD
	rm -r docs; hugo; cp -r public docs
=======
	hugo --destination docs
>>>>>>> gh-page
