# Bundled fonts

Only the Material Icons font is served from this directory; the application
makes no external font requests. Text uses Roboto when the system provides it,
otherwise the platform system font (no text fonts are bundled).

- `material_icon.woff2`: Material Icons from Google `material-design-icons`,
  as shipped in the Model Explorer visualizer bundle (Apache-2.0,
  `LICENSE-material-icons.txt`).

`src/theme/typography.scss` declares the face;
`scripts/copy-license-notices.mjs` appends its license text to the built
`3rdpartylicenses.txt`.
