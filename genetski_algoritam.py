import numpy as np
import random
import matplotlib.pyplot as plt
import statistics

# Dimenzije lavirinta
ROWS, COLS = 13, 13
matrix_putanja = 'dataset\\maze 13.txt'
end = (11,12)
#(maze, start, end, pop_size=100, generations=1000, mutation_rate=0.1, move_count=60)

#ROWS, COLS = 17, 17
#matrix_putanja = 'dataset\\maze 12.txt'
#end = (1,11)#15,16
#(maze, start, end, pop_size=100, generations=1000, mutation_rate=0.1, move_count=60)

#ROWS, COLS = 21, 21
#matrix_putanja = 'dataset\\train\\data_y\\maze 8.txt'
#end = (19,20)
#(maze, start, end, pop_size=200, generations=3000, mutation_rate=0.1, move_count=150):

start = (1,0) #vrsta,kolona
# Pravci kretanja (gore, dole, levo, desno)
MOVES = ['U', 'D', 'L', 'R']
MOVE_MAP = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1), 'N':(0,0)}

def load_matrix(matrix_path):
    niz_brojeva = []
    matrica = []
    with open(matrix_path, 'r') as file:
        matrix = [list(map(int, line.strip().split(','))) for line in file.readlines()]
    for i in range(ROWS):
       for j in range(COLS):
           broj = matrix[i][j]
           niz_brojeva.append(broj)
       matrica.append(niz_brojeva)
       niz_brojeva = []

    print(np.array(matrica))
    #print(np.array(matrica)[1][0])#v,k
    return np.array(matrica)

# Funkcija koja proverava da li je pozicija u okviru lavirinta i nije zid
def is_valid_position(maze, position):
    x, y = position
    return 0 <= x < ROWS and 0 <= y < COLS and maze[x][y] == 0

# Funkcija za kretanje kroz lavirint prema sekvenci instrukcija
def apply_moves(start, moves, maze, end):
    duzina = len(moves)
    pravi_potez = [0]*duzina
    x, y = start
    visited = set()  # Prati posete
    penalties = 0
    steps_taken = 0  # Praćenje koliko je koraka napravljeno
    indeks = 0
    for move in moves:
        if move not in MOVE_MAP:
            continue
        dx, dy = MOVE_MAP[move]
        new_x, new_y = x + dx, y + dy

        if is_valid_position(maze, (new_x, new_y)):
            
            steps_taken += 1  # Pravi validan korak
            if (new_x, new_y) in visited:
                penalties += 5  # Kazna za vraćanje na isto mesto
            else:
                pravi_potez[indeks] = 1  
            x, y = new_x, new_y
            visited.add((x, y))
            if (x, y) == end:
                return x, y, penalties, pravi_potez
        else:
            penalties += 10  # Kazna za udaranje u zid

        indeks = indeks + 1

    return x, y, penalties, pravi_potez

def fitness_function(moves, maze, start, end):
    x, y, penalties, _ = apply_moves(start, moves, maze, end)
    distance = abs(x - end[0]) + abs(y - end[1])  # Manhattan udaljenost
    # Ako smo stigli do cilja, vraćamo najbolji mogući fitness
    # Nagrada za postizanje cilja
    if (x, y) == end:
        return 1000 - penalties 
    # Kazne za udaranje u zidove i vraćanje na iste pozicije
    penalties_factor = penalties   # Kazna za loše poteze
    # Fitness funkcija: kombinacija udaljenosti i kazni
    return -(distance + penalties_factor)  # Kazna + udaljenost


# Kreiranje inicijalne populacije
def create_population(pop_size, move_count):
    return [''.join(random.choices(MOVES, k=move_count)) for _ in range(pop_size)]

# Selekcija najboljih pojedinaca (elitizam)
def selection(population, fitness_scores, num_best):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:num_best]
#nova selekcija
def selection2(population, fitness_scores, num_best):
    list_boljih = []
    list_scors = []
    for i in range(num_best):
        maxx = max(fitness_scores)
        indexx = fitness_scores.index(maxx)
        list_boljih.append(population[indexx])
        list_scors.append(maxx)
        fitness_scores[indexx] = -10000
    return list_boljih, list_scors

# Ukrštanje (kombinacija puteva)
def crossover(parent1, parent2):
    point1 = random.randint(1, len(parent1) // 2)
    point2 = random.randint(point1, len(parent1) - 1)
    child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    return child
def crossover2(parent1, parent2, start,maze,end,score1, score2):
    _, _, _, pravi_potez1 = apply_moves(start,parent1,maze,end)
    _, _, _, pravi_potez2 = apply_moves(start,parent2,maze,end)
    child = ['R'] * len(pravi_potez2)
    for i in range(len(pravi_potez1)):
      if pravi_potez1[i] == pravi_potez2[i]:
         if score1 > score2:
             child[i] = parent1[i]
         else:
             child[i] = parent2[i]
      if pravi_potez1[i] == 1 and pravi_potez2[i] == 0:
         child[i] = parent1[i]
      if pravi_potez1[i] == 0 and pravi_potez2[i] == 1:
         child[i] = parent2[i]
    return child
def crossover3(parent1, parent2, start,maze,end,score1, score2):
    _, _, _, pravi_potez1 = apply_moves(start,parent1,maze,end)
    _, _, _, pravi_potez2 = apply_moves(start,parent2,maze,end)
    #child = ''.join(random.choices(MOVES, k=len(pravi_potez1))) 
    child = random.choices(MOVES, k=len(pravi_potez1))
    #print(child)
    index = 0
    for i in range(len(pravi_potez1)):
      if pravi_potez1[i] == 1 and pravi_potez2[i] == 1:
         if score1 > score2:
             child[index] = parent1[i]
             index += 1   
         else:
             child[index] = parent2[i]
             index += 1 
      if pravi_potez1[i] == 1 and pravi_potez2[i] == 0:
         child[index] = parent1[i]
         index += 1 
      if pravi_potez1[i] == 0 and pravi_potez2[i] == 1:
         child[index] = parent2[i]
         index += 1 
    #print(child)
    return child

# Mutacija (promena slučajnog pravca)
def mutate1(individual, mutation_rate):
    mutated = list(individual)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.choice(MOVES)
    return ''.join(mutated)
def mutate(individual, mutation_rate):
    mutated = individual
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.choice(MOVES)
    return mutated
def mutate2(individual, mutation_rate, maze, start, end):
    mutated = list(individual)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.choice(MOVES)
            # Dodaj uslov da mutacija preferira poteze ka cilju
            x, y, _, _ = apply_moves(start, mutated[:i], maze,end)
            if abs(x - end[0]) + abs(y - end[1]) > abs(start[0] - end[0]) + abs(start[1] - end[1]):
                mutated[i] = random.choice([m for m in MOVES if is_valid_position(maze, (x + MOVE_MAP[m][0], y + MOVE_MAP[m][1]))])
    return ''.join(mutated)

# Glavna funkcija genetskog algoritma
def genetic_algorithm(maze, start, end, pop_size=100, generations=1500, mutation_rate=0.1, move_count=100):
    population = create_population(pop_size, move_count)
    
    for gen in range(generations):
        fitness_scores = [fitness_function(ind, maze, start, end) for ind in population]
        #srednja vrednost fit funkcije svake generacije
        #print(statistics.mean(fitness_scores))
        # Ako neki pojedinac stigne do izlaza
        if max(fitness_scores) == 1000:
            best_individual = population[fitness_scores.index(1000)]
            print(fitness_function(best_individual, maze, start, end))
            print(f"Rešenje pronađeno u generaciji {gen}: {best_individual}")
            _, _, _, pravi_potez = apply_moves(start,best_individual,maze,end)
            print(pravi_potez)
            return best_individual

        # Selekcija najboljih
        #selected = selection(population, fitness_scores, pop_size // 20)
        selected,scores = selection2(population, fitness_scores, pop_size // 10)

        # Kreiranje nove populacije kroz ukrštanje i mutaciju
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            #child = crossover(parent1, parent2)
            child = crossover3(parent1, parent2,start,maze,end,scores[selected.index(parent1)],scores[selected.index(parent2)])
            child = mutate(child, mutation_rate)
            #child = mutate2(child, mutation_rate,maze,start,end)
            new_population.append(child)
     
        population = new_population

    print("Rešenje nije pronađeno.")
    return None


maze = load_matrix(matrix_putanja)

plt.figure(figsize=(5,5))
plt.imshow(maze, cmap='gray')
plt.text(start[1], start[0], '@', ha='center', va='center', color='red', fontsize=20)
plt.xticks([]), plt.yticks([])
plt.grid(color='black', linewidth=2)
plt.show()

# Pokretanje genetskog algoritma
best_path = genetic_algorithm(maze, start, end)
# Ispis rezultata
if best_path:
    print(f"Najbolji pronađeni put: {best_path}")
    start_tmp = start
    plt.figure(figsize=(5,5))
    plt.imshow(maze, cmap='gray')
    plt.text(start[1], start[0], '@', ha='center', va='center', color='red', fontsize=20)
    for korak in best_path:
      x,y = start_tmp
      mx,my = MOVE_MAP[korak]
      nx,ny = x + mx, y + my
      plt.text(ny, nx, '@', ha='center', va='center', color='red', fontsize=20)
      if (nx,ny) == end:
          break
      start_tmp = (nx,ny)
    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()
else:
    print("Algoritam nije uspeo da pronađe rešenje.")
