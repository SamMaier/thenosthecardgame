const fs = require("fs/promises");
const path = require("path");
const sharp = require("sharp");

const glyphDirectory = __dirname + path.sep + "glyphs";
const pngDirectory = glyphDirectory + path.sep + "png";

function getAttribute(attributes, name) {
  const match = attributes.match(new RegExp(`\\b${name}\\s*=\\s*(["'])(.*?)\\1`));
  return match ? match[2] : undefined;
}

function getSvgRoot(svg) {
  const match = svg.match(/<svg\b([^>]*)>([\s\S]*?)<\/svg>/i);
  if (!match) {
    throw new Error("Expected an SVG document with an opening and closing <svg> tag");
  }
  return { attributes: match[1], content: match[2] };
}

async function expandSvgReferences(svg, sourcePath, ancestors = new Set()) {
  const source = path.resolve(sourcePath);
  if (ancestors.has(source)) {
    throw new Error(`Circular SVG reference: ${source}`);
  }

  const nextAncestors = new Set(ancestors);
  nextAncestors.add(source);
  const references = [...svg.matchAll(/<use\b([^>]*?)\/>/gi)];

  for (const reference of references) {
    const attributes = reference[1];
    const href = getAttribute(attributes, "href") ?? getAttribute(attributes, "xlink:href");
    if (!href || path.extname(href).toLowerCase() !== ".svg") {
      continue;
    }

    const referencedPath = path.resolve(path.dirname(source), href);
    const relativePath = path.relative(glyphDirectory, referencedPath);
    if (relativePath.startsWith(`..${path.sep}`) || relativePath === ".." || path.isAbsolute(relativePath)) {
      throw new Error(`SVG references must stay inside ${glyphDirectory}: ${href}`);
    }

    const referencedSvg = await fs.readFile(referencedPath, "utf8");
    const { attributes: rootAttributes, content } = getSvgRoot(referencedSvg);
    const viewBox = getAttribute(rootAttributes, "viewBox");
    if (!viewBox) {
      throw new Error(`Referenced SVG has no viewBox: ${referencedPath}`);
    }

    const x = getAttribute(attributes, "x") ?? "0";
    const y = getAttribute(attributes, "y") ?? "0";
    const width = getAttribute(attributes, "width");
    const height = getAttribute(attributes, "height");
    const nestedAttributes = [
      `x="${x}"`,
      `y="${y}"`,
      width ? `width="${width}"` : "",
      height ? `height="${height}"` : "",
      `viewBox="${viewBox}"`,
    ].filter(Boolean).join(" ");
    const expandedContent = await expandSvgReferences(content, referencedPath, nextAncestors);
    svg = svg.replace(reference[0], `<svg ${nestedAttributes}>${expandedContent}</svg>`);
  }

  return svg;
}

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
      const sourcePath = path.join(glyphDirectory, svgFile);
      const sourceSvg = await fs.readFile(sourcePath, "utf8");
      const rasterSvg = await expandSvgReferences(sourceSvg, sourcePath);
      await sharp(Buffer.from(rasterSvg))
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
