for value in {0..99}
do
	./driver > save
	python3 ../../../evaluate/addTime.py save
done

