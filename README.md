
# Trip Planner

This project is a route and search API for planning trips across a city, built in DSSL2 (a Racket-based teaching language). It was developed as the final project for Northwestern University’s CS 214: Data Structures & Functional Programming.

## 🚀 Features

The Trip Planner supports three main types of queries:

1. **`locate-all(category)`**  
   Returns all positions where a point-of-interest (POI) of a given category exists.

2. **`plan-route(start, poi_name)`**  
   Returns the shortest path from a given starting location to a POI with the specified name.

3. **`find-nearby(start, category, n)`**  
   Finds the `n` closest POIs in the given category near the start location.

Each query adheres to a strict API interface and returns results as linked lists of raw positions or POIs.

## 🧭 Data Representation

The system models:

- **Positions:** `(latitude, longitude)`
- **Road segments:** Connect two positions; roads are two-way.
- **POIs:** Include name, category, and position; only located at road endpoints.

All input is given in "raw" vector formats, and outputs use standard `cons` lists.

## 🛠 Technologies and Design

- **Language:** DSSL2 (subset of Racket)
- **Data Structures:** Custom hash tables, adjacency matrices, binary heaps, and lists
- **Algorithms:** Dijkstra’s shortest path, nearest neighbor search, and basic filtering
- **Design Focus:** Modular internal representations, clean separation between raw input/output formats and internal structures

## 📄 File Overview

- `planner.rkt` – Main source file implementing the `TripPlanner` class and the full API
- `README.md` – Project overview (this file)

## 📚 Learning Objectives

- Apply abstract data types (ADTs) and algorithms to open-ended design
- Build and test complex systems with real-world modeling
- Navigate tradeoffs between implementation complexity and performance
- Consider ethical implications of algorithms and product features

## ✍️ Ethical Reflection (from assignment)

We considered potential misuses of the planner, such as biased ranking or shadow banning POIs, and wrote design documents evaluating the ethical consequences of such choices.

## 🧪 Testing

This project passed functional test suites across:

- Basic and advanced queries for `locate-all`
- Route planning under multiple path scenarios
- Nearby POI searches with edge cases and large datasets

## 📜 Honor Code

All code adheres to Northwestern’s academic integrity policy and was submitted with the required integrity declaration.

---

> Developed for CS 214: Data Structures and Functional Programming  
> Northwestern University, Spring 2025
