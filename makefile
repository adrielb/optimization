all:
	python -W ignore::DeprecationWarning test_spsa.py

clean:
	rm *.pyc
