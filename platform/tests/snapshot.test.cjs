const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'server.cjs'), 'utf8');
const snapshotCode = source.slice(source.indexOf('function platformStorageRoots()'), source.indexOf('async function snapshotFromUpload('));
const projectsCode = source.slice(source.indexOf('async function listProjects()'), source.indexOf('function resolveProgram('));
const constants = ['SKIP_DIRS', 'SKIP_EXT', 'MAX_CODE_FILE', 'MAX_CODE_TOTAL'].map(name => source.match(new RegExp(`^const ${name} = .+;$`, 'm'))[0]).join('\n');
const base = fs.mkdtempSync(path.join(os.tmpdir(), 'eeg-package-test-'));
async function put(root, rel, text = 'fixture') {
  const file = path.join(root, rel);
  await fsp.mkdir(path.dirname(file), { recursive: true });
  await fsp.writeFile(file, text);
}
async function files(root) {
  const result = [];
  async function walk(dir) {
    for (const e of await fsp.readdir(dir, { withFileTypes: true })) {
      const file = path.join(dir, e.name);
      if (e.isDirectory()) await walk(file); else result.push(path.relative(root, file));
    }
  }
  await walk(root);
  return result.sort();
}
async function scenario(name, runtimeRelative) {
  const src = path.join(base, name, 'EEG');
  const ROOT = path.join(src, 'platform');
  const RUNTIME = path.join(src, runtimeRelative);
  const SAMPLE_DIR = path.join(ROOT, 'samples');
  for (const [file, text] of Object.entries({
    'train.py': 'print("train")',
    'run.sh': '#!/bin/sh\npython train.py\n',
    'models/model.py': 'MODEL = True',
    'model.mpc': '# training program',
    'platform-extra/keep.py': '# similarly named directory stays',
    'config.yaml': 'batch: 4',
    'platform/config.local.json': '{"apiKey":"dummy-fixture-only"}',
    'platform/samples/leak_demo.py': '# should never be sent for review',
    'platform/samples/demo.mpc': '# only listed as a platform sample',
    '.git/config': '# should be skipped',
    '__pycache__/a.pyc': '# should be skipped',
    'weights/checkpoint.pt': '# should be skipped',
    'data/train.pt': '# should be skipped',
    'node_modules/package/index.js': '# should be skipped'
  })) await put(src, file, text);
  await fsp.chmod(path.join(src, 'run.sh'), 0o755);
  await put(RUNTIME, 'state/state.json');
  await put(RUNTIME, 'tasks/old/code/old.py');
  await put(RUNTIME, 'Player-Data/P0.key');
  await fsp.symlink(ROOT, path.join(src, 'platform-alias'));
  await fsp.symlink(RUNTIME, path.join(src, 'runtime-alias'));
  const alias = path.join(path.dirname(src), 'EEG-alias');
  await fsp.symlink(src, alias);
  const big = await fsp.open(path.join(src, 'large.txt'), 'w');
  await big.truncate(2 * 1024 * 1024 + 1);
  await big.close();
  const context = vm.createContext({ fs, fsp, path, ROOT, RUNTIME, SAMPLE_DIR, projectRoots: () => [alias], state: { datasets: [] } });
  vm.runInContext(constants + '\n' + snapshotCode + '\n' + projectsCode, context);
  const dst = path.join(RUNTIME, 'tasks/new/code');
  await context.snapshotFromPath(alias, dst);
  const copied = await files(dst);
  assert.deepEqual(copied, ['config.yaml', 'model.mpc', 'models/model.py', 'platform-extra/keep.py', 'run.sh', 'train.py']);
  assert.equal(await fsp.readFile(path.join(dst, 'train.py'), 'utf8'), 'print("train")');
  assert.equal((await fsp.stat(path.join(dst, 'run.sh'))).mode & 0o777, 0o755);
  await assert.rejects(context.snapshotFromPath(ROOT, path.join(base, 'rejected')), /不能作为训练项目/);
  await assert.rejects(context.snapshotFromPath(path.join(src, 'runtime-alias'), path.join(base, 'rejected2')), /不能作为训练项目/);
  const listing = JSON.parse(JSON.stringify(await context.listProjects()));
  assert.deepEqual(listing.projects[0].mpc, ['model.mpc']);
  assert.deepEqual(listing.projects[0].entries, ['train.py']);
  assert.deepEqual(listing.samples, ['demo.mpc']);
  console.log('PASS', name, 'nested snapshot, alias exclusion, normal files and executable mode, existing skips, root rejection, project listing');
}
(async () => {
  try {
    await scenario('default-runtime', 'platform/runtime');
    await scenario('custom-runtime', 'task-storage');
  } finally { await fsp.rm(base, { recursive: true, force: true }); }
})().catch(error => { console.error(error); process.exitCode = 1; });
