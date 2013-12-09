all:
	python -W ignore::DeprecationWarning test_spsa_rprop.py

clean:
	rm *.pyc
