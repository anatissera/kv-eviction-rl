const fs = require('fs');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  console.error('Usage: node ua-arch-analyze.js <input.json> <output.json>');
  process.exit(1);
}

let input;
try {
  input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  console.error('Failed to parse input JSON:', e.message);
  process.exit(1);
}

const { fileNodes, importEdges, allEdges } = input;

// A. Directory Grouping
// Find common prefix
function getPathParts(filePath) {
  return filePath.split('/').filter(p => p !== '');
}

const allPaths = fileNodes.map(n => n.filePath);

// Common prefix
let commonPrefix = '';
const splitPaths = allPaths.map(getPathParts);
if (splitPaths.length > 0) {
  const first = splitPaths[0];
  let prefixLen = 0;
  for (let i = 0; i < first.length; i++) {
    if (splitPaths.every(p => p[i] === first[i])) {
      prefixLen = i + 1;
    } else {
      break;
    }
  }
  // Check if all start with 'src'
  if (splitPaths.every(p => p[0] === 'src')) {
    commonPrefix = 'src/';
  }
}

// Group by first dir after common prefix
const directoryGroups = {};
for (const node of fileNodes) {
  const parts = getPathParts(node.filePath);
  let groupKey;

  if (commonPrefix === 'src/') {
    // After 'src/', the next segment
    if (parts[0] === 'src' && parts.length > 1) {
      groupKey = parts[1]; // e.g., 'kvcompression'
    } else {
      groupKey = parts[0] || 'root';
    }
  } else {
    groupKey = parts[0] || 'root';
  }

  if (!directoryGroups[groupKey]) directoryGroups[groupKey] = [];
  directoryGroups[groupKey].push(node.id);
}

// More fine-grained grouping based on actual subdirectory structure
// Since all src files are under src/kvcompression/, group by their next level
const fineGroups = {};
for (const node of fileNodes) {
  const parts = getPathParts(node.filePath);
  let groupKey;

  if (parts[0] === 'src' && parts[1] === 'kvcompression') {
    if (parts.length === 3) {
      // Direct file in src/kvcompression/
      groupKey = 'kvcompression_root';
    } else {
      groupKey = parts[2]; // subdirectory name
    }
  } else if (parts[0] === 'tests') {
    groupKey = 'tests';
  } else if (parts[0] === 'configs') {
    groupKey = 'configs';
  } else if (parts[0] === 'scripts') {
    groupKey = 'scripts';
  } else if (parts[0] === 'notebooks') {
    groupKey = 'notebooks';
  } else if (parts[0] === 'assets') {
    groupKey = 'assets';
  } else if (parts[0] === '.understand-anything') {
    groupKey = 'tooling';
  } else {
    groupKey = 'root';
  }

  if (!fineGroups[groupKey]) fineGroups[groupKey] = [];
  fineGroups[groupKey].push(node.id);
}

// B. Node Type Grouping
const nodeTypeGroups = {};
for (const node of fileNodes) {
  const t = node.type;
  if (!nodeTypeGroups[t]) nodeTypeGroups[t] = [];
  nodeTypeGroups[t].push(node.id);
}

// C. Import adjacency
const fanOut = {};
const fanIn = {};
for (const node of fileNodes) {
  fanOut[node.id] = 0;
  fanIn[node.id] = 0;
}
for (const edge of importEdges) {
  if (fanOut[edge.source] !== undefined) fanOut[edge.source]++;
  if (fanIn[edge.target] !== undefined) fanIn[edge.target]++;
}

// D. Cross-category dependency analysis
const crossCategoryMap = {};
for (const edge of allEdges) {
  const srcNode = fileNodes.find(n => n.id === edge.source);
  const tgtNode = fileNodes.find(n => n.id === edge.target);
  if (!srcNode || !tgtNode) continue;
  const key = `${srcNode.type}->${tgtNode.type}:${edge.type}`;
  crossCategoryMap[key] = (crossCategoryMap[key] || 0) + 1;
}
const crossCategoryEdges = Object.entries(crossCategoryMap).map(([key, count]) => {
  const [types, edgeType] = key.split(':');
  const [fromType, toType] = types.split('->');
  return { fromType, toType, edgeType, count };
});

// E. Inter-group import frequency (using fine groups)
function getGroupForNode(nodeId) {
  for (const [grp, ids] of Object.entries(fineGroups)) {
    if (ids.includes(nodeId)) return grp;
  }
  return 'unknown';
}

const interGroupMap = {};
for (const edge of importEdges) {
  const srcGrp = getGroupForNode(edge.source);
  const tgtGrp = getGroupForNode(edge.target);
  if (srcGrp === tgtGrp) continue;
  const key = `${srcGrp}->${tgtGrp}`;
  interGroupMap[key] = (interGroupMap[key] || 0) + 1;
}
const interGroupImports = Object.entries(interGroupMap).map(([key, count]) => {
  const [from, to] = key.split('->');
  return { from, to, count };
}).sort((a, b) => b.count - a.count);

// F. Intra-group density
const intraGroupDensity = {};
for (const [grp, ids] of Object.entries(fineGroups)) {
  const idSet = new Set(ids);
  let internalEdges = 0;
  let totalEdges = 0;
  for (const edge of importEdges) {
    const srcIn = idSet.has(edge.source);
    const tgtIn = idSet.has(edge.target);
    if (srcIn || tgtIn) totalEdges++;
    if (srcIn && tgtIn) internalEdges++;
  }
  intraGroupDensity[grp] = {
    internalEdges,
    totalEdges,
    density: totalEdges > 0 ? internalEdges / totalEdges : 0
  };
}

// G. Directory pattern matching
const patternMap = {
  routes: 'api', api: 'api', controllers: 'api', endpoints: 'api', handlers: 'api',
  services: 'service', core: 'service', lib: 'service', domain: 'service', logic: 'service',
  models: 'data', db: 'data', data: 'data', persistence: 'data', repository: 'data', entities: 'data',
  components: 'ui', views: 'ui', pages: 'ui', ui: 'ui', layouts: 'ui', screens: 'ui',
  middleware: 'middleware', plugins: 'middleware', interceptors: 'middleware', guards: 'middleware',
  utils: 'utility', helpers: 'utility', common: 'utility', shared: 'utility', tools: 'utility',
  config: 'config', constants: 'config', env: 'config', settings: 'config', configs: 'config',
  __tests__: 'test', test: 'test', tests: 'test', spec: 'test', specs: 'test',
  types: 'types', interfaces: 'types', schemas: 'types', contracts: 'types', dtos: 'types',
  hooks: 'middleware',
  store: 'state', state: 'state', reducers: 'state', actions: 'state', slices: 'state',
  assets: 'assets', static: 'assets', public: 'assets',
  migrations: 'data',
  bin: 'entry', cmd: 'entry',
  docs: 'documentation', documentation: 'documentation', wiki: 'documentation',
  deploy: 'infrastructure', deployment: 'infrastructure', infra: 'infrastructure', infrastructure: 'infrastructure',
  k8s: 'infrastructure', kubernetes: 'infrastructure', helm: 'infrastructure',
  terraform: 'infrastructure', docker: 'infrastructure',
  scripts: 'utility',
  notebooks: 'documentation',
  entrypoints: 'entry',
  presses: 'service',
  rl: 'service',
  kv_cache: 'service',
  metrics: 'utility',
  kvcompression_root: 'entry',
  tooling: 'config',
  root: 'documentation'
};

const patternMatches = {};
for (const grp of Object.keys(fineGroups)) {
  patternMatches[grp] = patternMap[grp] || 'service';
}

// H. Deployment topology
const infraFiles = fileNodes
  .filter(n => {
    const p = n.filePath.toLowerCase();
    return p.includes('dockerfile') || p.includes('docker-compose') ||
           p.endsWith('.tf') || p.endsWith('.tfvars') ||
           p.includes('.github/workflows') || p.includes('.gitlab-ci') ||
           p === 'makefile' || p.includes('k8s') || p.includes('kubernetes');
  })
  .map(n => n.filePath);

const deploymentTopology = {
  hasDockerfile: infraFiles.some(f => f.toLowerCase().includes('dockerfile')),
  hasCompose: infraFiles.some(f => f.toLowerCase().includes('docker-compose')),
  hasK8s: infraFiles.some(f => f.toLowerCase().includes('k8s') || f.toLowerCase().includes('kubernetes')),
  hasTerraform: infraFiles.some(f => f.endsWith('.tf')),
  hasCI: infraFiles.some(f => f.includes('.github/workflows') || f.includes('.gitlab-ci')),
  infraFiles
};

// I. Data pipeline detection
const dataPipeline = {
  schemaFiles: fileNodes.filter(n => n.filePath.match(/\.(sql|graphql|gql|proto|prisma)$/)).map(n => n.filePath),
  migrationFiles: fileNodes.filter(n => n.filePath.includes('migration')).map(n => n.filePath),
  dataModelFiles: fileNodes.filter(n => n.tags.includes('dataset') || n.tags.includes('data')).map(n => n.filePath),
  apiHandlerFiles: fileNodes.filter(n => n.tags.includes('entrypoint')).map(n => n.filePath)
};

// J. Documentation coverage
const groupsWithDocs = Object.keys(fineGroups).filter(grp => {
  const ids = fineGroups[grp];
  return ids.some(id => {
    const node = fileNodes.find(n => n.id === id);
    return node && (node.type === 'document' || node.name.endsWith('.md'));
  });
});
const docCoverage = {
  groupsWithDocs: groupsWithDocs.length,
  totalGroups: Object.keys(fineGroups).length,
  coverageRatio: groupsWithDocs.length / Object.keys(fineGroups).length,
  undocumentedGroups: Object.keys(fineGroups).filter(g => !groupsWithDocs.includes(g))
};

// K. Dependency direction
const dependencyDirection = interGroupImports
  .filter(e => e.count > 0)
  .map(e => ({ dependent: e.from, dependsOn: e.to }));

// File stats
const filesPerGroup = {};
for (const [grp, ids] of Object.entries(fineGroups)) {
  filesPerGroup[grp] = ids.length;
}
const nodeTypeCounts = {};
for (const [t, ids] of Object.entries(nodeTypeGroups)) {
  nodeTypeCounts[t] = ids.length;
}

const output = {
  scriptCompleted: true,
  directoryGroups: fineGroups,
  nodeTypeGroups,
  crossCategoryEdges,
  interGroupImports,
  intraGroupDensity,
  patternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats: {
    totalFileNodes: fileNodes.length,
    filesPerGroup,
    nodeTypeCounts
  },
  fileFanIn: fanIn,
  fileFanOut: fanOut
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log('Analysis complete. Total nodes:', fileNodes.length);
  console.log('Groups:', Object.keys(fineGroups).join(', '));
} catch (e) {
  console.error('Failed to write output:', e.message);
  process.exit(1);
}
