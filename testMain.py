import sys

if __name__ == '__main__':
	filename = 'pokemon/025.png'
	pokemon = 'Pikachu'

	f = open('updates.txt', 'w')
	f.write(filename+'\n')
	f.write(pokemon)
	f.close()
	sys.stdout.flush()
