## Tailwind setup & troubleshooting

If Tailwind isn't being applied in your local dev build, follow these steps to make sure your environment is configured correctly.

1. Install or update dependencies (run from the `frontend` folder):

```powershell
cd frontend
npm install --save-dev tailwindcss@^3.4.24 postcss@^8.4.24 autoprefixer@^10.4.14
```

2. If you don't already have config files, create them with the Tailwind CLI (optional).

Recommended (modern npm):

```powershell
# install as dev-dependencies first
npm install --save-dev tailwindcss@^3.4.18 postcss@^8.4.24 autoprefixer@^10.4.14

# use npm exec or npx to run the CLI
npm exec -- tailwindcss init -p
# OR (if `npx tailwindcss init -p` errors):
npx -p tailwindcss@3 tailwindcss init -p
```

This creates `tailwind.config.js` and `postcss.config.js`. This repo already includes both.

3. Confirm you have the Tailwind directives in `src/index.css` (this repo already has them):

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

4. Confirm your entry `src/main.jsx` imports the CSS file (this repo already does):

```js
import './index.css'
```

5. Run the dev server:

```powershell
npm run dev
```

6. If Tailwind classes are not showing up, check these common issues:
- Make sure `tailwind.config.js` `content` paths include `./index.html` and `./src/**/*.{js,jsx,ts,tsx}`.
- If using Windows PowerShell, ensure you re-start the dev server after changing configs or dependencies.
- If you see a PostCSS or tailwind error during build, try making `postcss.config.js` a CommonJS export:

```js
module.exports = { plugins: { tailwindcss: {}, autoprefixer: {} } }
```

If you continue to have trouble, paste the full npm install / npm run dev error output here and I'll help debug.
