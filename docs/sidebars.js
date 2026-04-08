/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  main: [
    'index',
    'bundles',
    'getting-started',
    'beginner-guide',
    'schema',
    'preprocess-postprocess',
    'bundles',
    'runtime',
    'pipelines',
    'ui',
    'cli',
    {
      type: 'category',
      label: 'Examples',
      items: ['examples/mnist', 'examples/inception-v3']
    },
    'faq',
    'roadmap'
  ]
};

module.exports = sidebars;
