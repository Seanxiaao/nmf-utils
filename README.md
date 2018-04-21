# nmf-utils
codes relevent with my undergraduate thesis, however the codes is still being revised.

I have implemented the algorithme of nmf, semi-nmf, semi-nmf with constraints, convex-nmf, and kernel-nmf.

todo-list: implementing the algorithm to boost the iteration speed.

TOUSE: 
   first get into the graduatethesis folder, methods may differ for your settings 

	cd ~/downloads/nmf-utils/graduateThesis
	
   use nmf for your data
   
    python3 main.py -d "yourdata.txt"
    
   make the test
    
    python3 main.py -t 
    
    

TOMODIEFY:
	some part of the code are boosting by cython, if you need to modified my code, be sure to make the compilation.
	  
	  python3 setup.py build_ext --inplace
	
#####the code is still incomplete, some error may occured