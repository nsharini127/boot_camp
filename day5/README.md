📘 Day 05 — Modern Frontend Setup (React + TypeScript)
🎯 Core Goal

Set up a modern frontend environment using React + TypeScript, understand typed components, and build a simple PredictionForm that captures user inputs and displays a mock prediction.
This prepares the foundation for backend–frontend integration on Day 06 and Day 07.

🧠 Reading / Theory / Topics to Cover

Before jumping into code, review the core concepts for today:

1. React Component Model

What components are

How state works

How user inputs update component state

Declarative UI model

2. TypeScript Basics (for React)

type vs interface

Typed state and props

Why TypeScript helps avoid frontend errors

3. Vite (Project Setup Tool)

Why Vite is used for modern React projects

How Vite scaffolds a React+TS template

Running and building Vite projects

🎥 Useful YouTube Resources

Every React Concept Explained in 12 Minutes
https://youtu.be/wIyHSOugGGw?si=asMBHl_9ptgKHORV

TypeScript Crash Course
https://youtu.be/d56mG7DezGs?si=5peWVNxuCX1zSCUU

Vite Project Setup
https://youtu.be/KCrXgy8qtjM?si=aUTt9CFVVsbkPAuQ

📝 Problem Statement

Build the UI layer of the project:
A simple React + TypeScript app with a form that accepts:

A text input

A numeric threshold slider

A checkbox

…and displays a mock prediction result on submit.

This demonstrates:

Controlled components

Typed React state

Handling form submissions

Simple fetch workflow (mocked)

🛠️ Task 1 — Project Setup using Vite

Inside the repo root:

mkdir Day-05
cd Day-05
npm create vite@latest . -- --template react-ts
npm install
npm run dev


This starts the app at:
➡️ http://localhost:5173

🛠️ Task 2 — Create the Typed PredictionForm Component

Inside:

Day-05/src/components/PredictionForm.tsx


Implement:

Typed state

Text input, slider, checkbox

Mock fetch() submit handler

Loading indicator + result display

🛠️ Task 3 — Integrate Component into App

Modify:

Day-05/src/App.tsx


to display the PredictionForm on the page.

📂 Final Folder Structure

Your Day-05/ folder must include:

Day-05/
  README.md
  package.json
  tsconfig.json
  index.html
  vite.config.ts
  src/
    main.tsx
    App.tsx
    components/
      PredictionForm.tsx

💾 Submission Using Feature Branching

Follow the same workflow as Days 1–4.

1. Synchronize and Create Branch
git checkout main
git pull upstream main
git checkout -b feat/day-05-<your-name>

2. Commit Your Work
git add Day-05
git commit -m "feat(day05): React + TypeScript PredictionForm component"

3. Push to Origin
git push -u origin feat/day-05-<your-name>

4. Create Pull Request

From: your feature branch

To: main of georgeaaron7/bootycamp

Title:

[D05] <Your Name> - React + TypeScript PredictionForm


Description:

Added Day-05 React + TypeScript project.
Includes typed PredictionForm component, App integration, and mock submit flow.
Location: Day-05/