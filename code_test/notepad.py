
import time, sys

print('%s'%("hello world"), end='\r')

time.sleep(1)
sys.stdout.write("\033[K")

print('%s'%("bye now"))