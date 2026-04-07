/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  main: [
    'intro',
    'getting-started',
    'schema',
    'preprocess-postprocess',
    'bundles',
    'runtime',
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
