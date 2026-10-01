const fs = require("fs/promises");
const path = require("path");
const sharp = require("sharp");

const glyphDirectory = __dirname + path.sep + "glyphs";
const pngDirectory = glyphDirectory + path.sep + "png";

async function main() {
  await fs.mkdir(pngDirectory, { recursive: true });

  const entries = await fs.readdir(glyphDirectory, { withFileTypes: true });
  const svgFiles = entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".svg"))
    .map((entry) => entry.name)
    .sort();

  await Promise.all(
    svgFiles.map(async (svgFile) => {
      const pngFile = svgFile.replace(/\.svg$/i, ".png");
      await sharp(path.join(glyphDirectory, svgFile))
        .resize(240, 240)
        .png()
        .toFile(path.join(pngDirectory, pngFile));
    }),
  );

  console.log(`Rasterized ${svgFiles.length} glyphs into ${pngDirectory}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
